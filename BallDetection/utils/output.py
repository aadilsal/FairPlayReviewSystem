from typing import List, Dict, Any, Optional
from BallDetection.utils.config import DETECTION_CONFIG

def generate_output(ball_infos: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Produces final JSON/dict with each frame annotated:
    position, confidence_tier (High / Med / Low), source, uncertain flag
    
    Tier assignment:
    High: original YOLO anchor (conf >= config threshold)
    Med: YOLO rescue or CSRT-agreed
    Low: kinematic fallback or edge-suspected
    """
    output = {}
    conf_threshold = DETECTION_CONFIG.get('conf_threshold', 0.2)
    
    for i, info in enumerate(ball_infos):
        if info is None or info.get('ghost', False):
            continue
            
        source = info.get('source', 'yolo-anchor')
        # Handle case where old detections might not have a source explicitly set
        if source == 'yolo' or source is None:
            source = 'yolo-anchor'
            
        conf = info.get('conf', 0.0)
        
        # Determine position
        if 'interpolated_position' in info and info['interpolated_position'] is not None:
            pos = info['interpolated_position']
        else:
            box = info.get('box', [0.0, 0.0, 0.0, 0.0])
            pos = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
            
        # Determine confidence tier
        tier = 'Low'
        uncertain = False

        if source == 'yolo-anchor' and conf >= conf_threshold:
            tier = 'High'
        elif source in ['yolo-rescue', 'csrt-agreed', 'csrt-forward', 'csrt-backward']:
            tier = 'Med'
        elif source == 'kinematic':
            tier = 'Low'
            uncertain = True
        elif source == 'edge-suspected':
            # Promote to Med if intersection was solved parabolically (uncertain==False)
            if not info.get('uncertain', False):
                tier = 'Med'
                uncertain = False
            else:
                tier = 'Low'
                uncertain = True
        else:
            # Fallbacks exactly as described
            if conf >= conf_threshold:
                tier = 'High'
            elif conf > 0.0:
                tier = 'Med'
            else:
                tier = 'Low'
                uncertain = True

        output[str(i)] = {
            'frame_idx': info.get('frame_idx', i),
            'position': pos,
            'confidence_tier': tier,
            'source': source,
            'uncertain': uncertain
        }
        
    return {"frames": output}
